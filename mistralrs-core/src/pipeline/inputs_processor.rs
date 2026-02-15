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
        paged_attention::{KVCacheManager, _PAD_SLOT_ID},
        sequence::Sequence,
    };

    use super::{InputProcessorOutput, InputsProcessor, InputsProcessorType};

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

    pub struct PagedAttentionMeta {
        pub sliding_window: Option<usize>,
        pub block_size: usize,
        pub kv_cache_manager: Arc<tokio::sync::Mutex<KVCacheManager>>,
    }

    #[derive(Clone, Debug)]
    #[allow(dead_code)]
    pub struct PagedAttentionInputMetadata {
        /// Block tables, windowed when a global sliding_window is set.
        pub block_tables: Option<HashMap<DeviceLocation, Tensor>>,
        /// Context lens, capped by sliding_window when set.
        pub context_lens: Option<HashMap<DeviceLocation, Tensor>>,
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
        pub paged_kv_indptr: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_last_page_len: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_request_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_tile_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_o_indptr: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_chunk_size: Option<HashMap<DeviceLocation, Tensor>>,
        /// Number of cached tokens per sequence (from prefix cache hits).
        /// When present and > 0, gather_kv_cache + Sdpa is used during prefill
        /// instead of flash attention. The Q/K/V tensors should only contain
        /// the NEW (non-cached) tokens.
        pub num_cached_tokens: Option<Vec<usize>>,
        /// Number of new tokens per sequence (query lengths).
        pub query_lens: Option<Vec<usize>>,
        /// Cumulative query lengths [batch+1], u32 — for Sdpa varlen flash path.
        /// Precomputed to avoid Tensor::new in the forward hot path.
        pub cu_seqlens_q: Option<HashMap<DeviceLocation, Tensor>>,
        /// Cumulative KV lengths [batch+1], u32 — for gather_kv_cache and flash_attn_varlen.
        /// Each entry is sum of (cached + new) tokens.
        pub cu_seqlens_kv: Option<HashMap<DeviceLocation, Tensor>>,
    }

    impl PagedAttentionInputMetadata {
        /// Create a dummy input metadata, assuming that this will NOT be used for decoding.
        /// This is used for the case of imatrix generation.
        pub fn dummy(dev: &Device) -> candle_core::Result<Self> {
            Ok(PagedAttentionInputMetadata {
                block_tables: None,
                context_lens: None,
                max_context_len: None,
                full_block_tables: None,
                full_context_lens: None,
                full_max_context_len: None,
                slot_mappings: HashMap::from([(dev.location(), Tensor::new(&[0f32], dev)?)]),
                is_first_prompt_chunk: true,
                paged_kv_indptr: None,
                paged_kv_indices: None,
                paged_kv_last_page_len: None,
                paged_kv_request_indices: None,
                paged_kv_tile_indices: None,
                paged_kv_o_indptr: None,
                paged_kv_chunk_size: None,
                num_cached_tokens: None,
                query_lens: None,
                cu_seqlens_q: None,
                cu_seqlens_kv: None,
            })
        }
    }

    /// Flash attention sequence length metadata.
    ///
    /// `cumulative_seqlens_q/k` use **padded** lengths (each sequence is padded to
    /// `max_len` in the batch). This matches the padded Q/K tensors in the normal
    /// prefill and decode paths.
    ///
    /// For the **prefix cache path**, K/V are gathered from the paged cache into a
    /// packed (non-padded) layout via `gather_kv_cache`. The packed K/V lengths are
    /// given by `PagedAttentionInputMetadata::cu_seqlens_kv`, NOT by
    /// `cumulative_seqlens_k` here. The prefix cache attention call must build a
    /// local `FlashParams` that swaps in `cu_seqlens_kv` for K.
    #[derive(Clone, Debug)]
    pub struct FlashParams {
        pub max_q: u32,
        pub max_k: u32,
        pub cumulative_seqlens_q: HashMap<DeviceLocation, Tensor>,
        pub cumulative_seqlens_k: HashMap<DeviceLocation, Tensor>,
        pub causal: bool,
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
        let mut paged_attn_context_lens = Vec::new();
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
                // Padded lengths — see FlashParams doc comment for prefix cache nuance.
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

                // Block table covers the full context (cached + new)
                let table_for_seq = table.clone();

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
                paged_attn_context_lens.push(ctxt_len);
                block_tables.push(table_for_seq);
            }
        }

        let (max_q, max_k, seqlens_q_map, seqlens_k_map) = if flash_attn {
            // SAFETY: seqlens_q/k are initialized with vec![0] when flash_attn is true,
            // so they are guaranteed to be non-empty here.
            let max_q = *seqlens_q
                .iter()
                .max()
                .expect("seqlens_q should not be empty when flash_attn is enabled");
            let max_k = *seqlens_k
                .iter()
                .max()
                .expect("seqlens_k should not be empty when flash_attn is enabled");
            // Create tensors on CPU first to avoid CUDA context issues when copying
            // between different GPU devices. Each GPU has its own CUDA context, and
            // candle/cudarc doesn't properly switch contexts when doing GPU-to-GPU
            // transfers (which go through CPU). By creating on CPU first, we avoid
            // the cross-context memory access that causes CUDA_ERROR_INVALID_VALUE.
            let seqlens_q = Tensor::new(seqlens_q, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(DType::U32)?;
            let seqlens_k = Tensor::new(seqlens_k, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(DType::U32)?;

            let mut seqlens_q_map = HashMap::new();
            let mut seqlens_k_map = HashMap::new();

            let devices = mapper.unwrap().get_unique_devices();
            for device in devices {
                seqlens_q_map.insert(device.location(), seqlens_q.to_device(&device)?);
                seqlens_k_map.insert(device.location(), seqlens_k.to_device(&device)?);
            }
            (max_q, max_k, seqlens_q_map, seqlens_k_map)
        } else {
            (0, 0, HashMap::new(), HashMap::new())
        };

        let input = Tensor::cat(&seqs_tensors, 0).unwrap();

        let paged_attn_meta = if paged_attn_metadata.is_some() {
            // Create paged attention tensors on CPU first (see comment above about CUDA contexts)
            let max_slot_mapping_len = slot_mappings.iter().map(|x| x.len()).max().unwrap();
            let slot_mappings = _make_tensor_with_pad(
                slot_mappings,
                max_slot_mapping_len,
                _PAD_SLOT_ID,
                &Device::Cpu,
            )?;

            let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
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

            for device in devices {
                slot_mappings_map
                    .insert(device.location(), slot_mappings.clone().to_device(&device)?);
                block_tables_map
                    .insert(device.location(), block_tables.clone().to_device(&device)?);
                context_lens_map
                    .insert(device.location(), context_lens.clone().to_device(&device)?);
            }

            Some(PagedAttentionInputMetadata {
                slot_mappings: slot_mappings_map,
                block_tables: Some(block_tables_map),
                context_lens: Some(context_lens_map),
                max_context_len: Some(max_context_len),
                full_block_tables: None,
                full_context_lens: None,
                full_max_context_len: None,
                is_first_prompt_chunk: chunk_offset_toks == 0,
                paged_kv_indptr: None,
                paged_kv_indices: None,
                paged_kv_last_page_len: None,
                paged_kv_request_indices: None,
                paged_kv_tile_indices: None,
                paged_kv_o_indptr: None,
                paged_kv_chunk_size: None,
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
            flash_meta: FlashParams {
                max_k,
                max_q,
                cumulative_seqlens_k: seqlens_k_map,
                cumulative_seqlens_q: seqlens_q_map,
                causal: true,
            },
        })
    }

    fn make_completion_chunk<T: WithDType>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
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
        for (seq, ctxt) in input_seqs.iter().zip(toks) {
            let start_pos = ctxt.len().saturating_sub(1);
            let ctxt = ctxt[start_pos..].to_vec();
            seqlen_offsets.push(start_pos);
            context_lens.push((0, 1));
            position_ids.push(seq.len());

            if flash_attn {
                seqlens_q.push(ctxt.len() as u32);
                seqlens_k.push((ctxt.len() + start_pos) as u32);
            }

            seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let kv_mgr = get_mut_arcmutex!(paged_attn_metadata.kv_cache_manager);
                let table: Vec<usize> = kv_mgr
                    .get_block_ids(*seq.id())
                    .expect("Sequence must have allocated blocks for completion")
                    .to_vec();
                drop(kv_mgr);

                let block_pos = start_pos - seq.token_offset();
                let block_number = if block_pos / paged_attn_metadata.block_size >= table.len() {
                    panic!("Block table is too small (completion)! start_pos={} block_size={} table_len={}", block_pos, paged_attn_metadata.block_size, table.len());
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
                slot_mappings.push(vec![slot]);

                // Always collect the full (unwindowed) block tables.
                full_block_tables.push(table.clone());
                full_paged_attn_context_lens.push(seq.len());

                if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                    let window_start = seq.len().saturating_sub(sliding_window);
                    let slide_idx = window_start / paged_attn_metadata.block_size;
                    block_tables.push(table.get(slide_idx..).unwrap().to_vec());
                } else {
                    block_tables.push(table);
                }

                let paged_attn_context_len =
                    if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                        let window_start = seq.len().saturating_sub(sliding_window);
                        let block_aligned_start = (window_start / paged_attn_metadata.block_size)
                            * paged_attn_metadata.block_size;
                        seq.len() - block_aligned_start
                    } else {
                        seq.len()
                    };
                paged_attn_context_lens.push(paged_attn_context_len);
            }
        }

        let (max_q, max_k, seqlens_q_map, seqlens_k_map) = if flash_attn {
            // SAFETY: seqlens_q/k are initialized with vec![0] when flash_attn is true,
            // so they are guaranteed to be non-empty here.
            let max_q = *seqlens_q
                .iter()
                .max()
                .expect("seqlens_q should not be empty when flash_attn is enabled");
            let max_k = *seqlens_k
                .iter()
                .max()
                .expect("seqlens_k should not be empty when flash_attn is enabled");
            // Create tensors on CPU first to avoid CUDA context issues (see make_prompt_chunk)
            let seqlens_q = Tensor::new(seqlens_q, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(DType::U32)?;
            let seqlens_k = Tensor::new(seqlens_k, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(DType::U32)?;

            let mut seqlens_q_map = HashMap::new();
            let mut seqlens_k_map = HashMap::new();

            let devices = mapper.unwrap().get_unique_devices();
            for device in devices {
                seqlens_q_map.insert(device.location(), seqlens_q.to_device(&device)?);
                seqlens_k_map.insert(device.location(), seqlens_k.to_device(&device)?);
            }
            (max_q, max_k, seqlens_q_map, seqlens_k_map)
        } else {
            (0, 0, HashMap::new(), HashMap::new())
        };

        let paged_attn_meta = if let Some(paged_attn_input) = &paged_attn_metadata {
            // Create paged attention tensors on CPU first (see make_prompt_chunk for explanation)
            let slot_mappings =
                _make_tensor_with_pad(slot_mappings, 1, _PAD_SLOT_ID, &Device::Cpu)?;

            let max_block_table_len = block_tables
                .iter()
                .map(|x| x.len())
                .max()
                .expect("block_tables should not be empty when paged attention is enabled");

            let batch_size = block_tables.len();
            let mut paged_kv_indices = Vec::new();
            let mut paged_kv_indptr = Vec::with_capacity(batch_size + 1);
            let mut paged_kv_last_page_len = Vec::with_capacity(batch_size);
            paged_kv_indptr.push(0i32);
            let mut nnz_pages = 0i32;
            let block_size = paged_attn_input.block_size;
            for (table, context_len) in block_tables.iter().zip(paged_attn_context_lens.iter()) {
                let num_blocks = table.len();
                nnz_pages += num_blocks as i32;
                paged_kv_indptr.push(nnz_pages);
                paged_kv_indices.extend(table.iter().map(|x| *x as i32));
                let last_page_len = if num_blocks == 0 {
                    0usize
                } else {
                    let consumed = (num_blocks - 1) * block_size;
                    if *context_len < consumed {
                        panic!(
                            "paged kv context len underflow: context_len={} consumed={}",
                            context_len, consumed
                        );
                    }
                    *context_len - consumed
                };
                paged_kv_last_page_len.push(last_page_len as i32);
            }

            let request_indices: Vec<i32> = (0..batch_size as i32).collect();
            let kv_tile_indices = vec![0i32; batch_size];
            let o_indptr: Vec<i32> = (0..=batch_size as i32).collect();
            let kv_chunk_size = vec![block_size as i32];

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

            let paged_kv_indptr =
                Tensor::from_vec(paged_kv_indptr, (batch_size + 1,), &Device::Cpu)?;
            let paged_kv_indices =
                Tensor::from_vec(paged_kv_indices, (nnz_pages as usize,), &Device::Cpu)?;
            let paged_kv_last_page_len =
                Tensor::from_vec(paged_kv_last_page_len, (batch_size,), &Device::Cpu)?;
            let request_indices = Tensor::from_vec(request_indices, (batch_size,), &Device::Cpu)?;
            let kv_tile_indices = Tensor::from_vec(kv_tile_indices, (batch_size,), &Device::Cpu)?;
            let o_indptr = Tensor::from_vec(o_indptr, (batch_size + 1,), &Device::Cpu)?;
            let kv_chunk_size = Tensor::from_vec(kv_chunk_size, (1,), &Device::Cpu)?;

            // Build full (unwindowed) block tables and context lens.
            let full_max_block_table_len =
                full_block_tables.iter().map(|x| x.len()).max().unwrap_or(0);

            let full_block_tables_tensor = _make_tensor_with_pad(
                full_block_tables
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                full_max_block_table_len.max(1),
                0,
                &Device::Cpu,
            )?;
            let full_block_tables_tensor =
                full_block_tables_tensor.reshape(((), full_max_block_table_len.max(1)))?;

            let full_max_context_len = full_paged_attn_context_lens
                .iter()
                .max()
                .copied()
                .unwrap_or(0);

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
            let mut paged_kv_request_indices_map = HashMap::new();
            let mut paged_kv_tile_indices_map = HashMap::new();
            let mut paged_kv_o_indptr_map = HashMap::new();
            let mut paged_kv_chunk_size_map = HashMap::new();

            for device in devices {
                slot_mappings_map
                    .insert(device.location(), slot_mappings.clone().to_device(&device)?);
                block_tables_map
                    .insert(device.location(), block_tables.clone().to_device(&device)?);
                context_lens_map
                    .insert(device.location(), context_lens.clone().to_device(&device)?);
                full_block_tables_map.insert(
                    device.location(),
                    full_block_tables_tensor.clone().to_device(&device)?,
                );
                full_context_lens_map.insert(
                    device.location(),
                    full_context_lens_tensor.clone().to_device(&device)?,
                );
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
                paged_kv_request_indices_map.insert(
                    device.location(),
                    request_indices.clone().to_device(&device)?,
                );
                paged_kv_tile_indices_map.insert(
                    device.location(),
                    kv_tile_indices.clone().to_device(&device)?,
                );
                paged_kv_o_indptr_map
                    .insert(device.location(), o_indptr.clone().to_device(&device)?);
                paged_kv_chunk_size_map
                    .insert(device.location(), kv_chunk_size.clone().to_device(&device)?);
            }

            Some(PagedAttentionInputMetadata {
                slot_mappings: slot_mappings_map,
                block_tables: Some(block_tables_map),
                context_lens: Some(context_lens_map),
                max_context_len: Some(*max_context_len),
                full_block_tables: Some(full_block_tables_map),
                full_context_lens: Some(full_context_lens_map),
                full_max_context_len: Some(full_max_context_len),
                is_first_prompt_chunk: false,
                paged_kv_indptr: Some(paged_kv_indptr_map),
                paged_kv_indices: Some(paged_kv_indices_map),
                paged_kv_last_page_len: Some(paged_kv_last_page_len_map),
                paged_kv_request_indices: Some(paged_kv_request_indices_map),
                paged_kv_tile_indices: Some(paged_kv_tile_indices_map),
                paged_kv_o_indptr: Some(paged_kv_o_indptr_map),
                paged_kv_chunk_size: Some(paged_kv_chunk_size_map),
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
            flash_meta: FlashParams {
                max_k,
                max_q,
                cumulative_seqlens_k: seqlens_k_map,
                cumulative_seqlens_q: seqlens_q_map,
                causal: true,
            },
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
        )
        .map(|inputs| InnerInputProcessorOutput {
            inputs,
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_completion_input<T: WithDType + std::fmt::Debug>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
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
            );
        }

        make_completion_chunk(toks, input_seqs, device, paged_attn_metadata, mapper).map(|inputs| {
            InnerInputProcessorOutput {
                inputs,
                seq_indices: (0..input_seqs.len()).collect(),
            }
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
            _: Option<Arc<dyn Any>>,
            mut paged_attn_metadata: Option<PagedAttentionMeta>,
            mapper: Option<&dyn DeviceMapper>,
        ) -> Result<InputProcessorOutput> {
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
